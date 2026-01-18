import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class ArffDecoder:
    """An ARFF decoder."""

    def __init__(self):
        """Constructor."""
        self._conversors = []
        self._current_line = 0

    def _decode_comment(self, s):
        """(INTERNAL) Decodes a comment line.

        Comments are single line strings starting, obligatorily, with the ``%``
        character, and can have any symbol, including whitespaces or special
        characters.

        This method must receive a normalized string, i.e., a string without
        padding, including the "\r
" characters.

        :param s: a normalized string.
        :return: a string with the decoded comment.
        """
        res = re.sub('^\\%( )?', '', s)
        return res

    def _decode_relation(self, s):
        """(INTERNAL) Decodes a relation line.

        The relation declaration is a line with the format ``@RELATION
        <relation-name>``, where ``relation-name`` is a string. The string must
        start with alphabetic character and must be quoted if the name includes
        spaces, otherwise this method will raise a `BadRelationFormat` exception.

        This method must receive a normalized string, i.e., a string without
        padding, including the "\r
" characters.

        :param s: a normalized string.
        :return: a string with the decoded relation name.
        """
        _, v = s.split(' ', 1)
        v = v.strip()
        if not _RE_RELATION.match(v):
            raise BadRelationFormat()
        res = str(v.strip('"\''))
        return res

    def _decode_attribute(self, s):
        """(INTERNAL) Decodes an attribute line.

        The attribute is the most complex declaration in an arff file. All
        attributes must follow the template::

             @attribute <attribute-name> <datatype>

        where ``attribute-name`` is a string, quoted if the name contains any
        whitespace, and ``datatype`` can be:

        - Numerical attributes as ``NUMERIC``, ``INTEGER`` or ``REAL``.
        - Strings as ``STRING``.
        - Dates (NOT IMPLEMENTED).
        - Nominal attributes with format:

            {<nominal-name1>, <nominal-name2>, <nominal-name3>, ...}

        The nominal names follow the rules for the attribute names, i.e., they
        must be quoted if the name contains whitespaces.

        This method must receive a normalized string, i.e., a string without
        padding, including the "\r
" characters.

        :param s: a normalized string.
        :return: a tuple (ATTRIBUTE_NAME, TYPE_OR_VALUES).
        """
        _, v = s.split(' ', 1)
        v = v.strip()
        m = _RE_ATTRIBUTE.match(v)
        if not m:
            raise BadAttributeFormat()
        name, type_ = m.groups()
        name = str(name.strip('"\''))
        if type_[:1] == '{' and type_[-1:] == '}':
            try:
                type_ = _parse_values(type_.strip('{} '))
            except Exception:
                raise BadAttributeType()
            if isinstance(type_, dict):
                raise BadAttributeType()
        else:
            type_ = str(type_).upper()
            if type_ not in ['NUMERIC', 'REAL', 'INTEGER', 'STRING']:
                raise BadAttributeType()
        return (name, type_)

    def _decode(self, s, encode_nominal=False, matrix_type=DENSE):
        """Do the job the ``encode``."""
        self._current_line = 0
        if isinstance(s, str):
            s = s.strip('\r\n ').replace('\r\n', '\n').split('\n')
        obj: ArffContainerType = {'description': '', 'relation': '', 'attributes': [], 'data': []}
        attribute_names = {}
        data = _get_data_object_for_decoding(matrix_type)
        STATE = _TK_DESCRIPTION
        s = iter(s)
        for row in s:
            self._current_line += 1
            row = row.strip(' \r\n')
            if not row:
                continue
            u_row = row.upper()
            if u_row.startswith(_TK_DESCRIPTION) and STATE == _TK_DESCRIPTION:
                obj['description'] += self._decode_comment(row) + '\n'
            elif u_row.startswith(_TK_RELATION):
                if STATE != _TK_DESCRIPTION:
                    raise BadLayout()
                STATE = _TK_RELATION
                obj['relation'] = self._decode_relation(row)
            elif u_row.startswith(_TK_ATTRIBUTE):
                if STATE != _TK_RELATION and STATE != _TK_ATTRIBUTE:
                    raise BadLayout()
                STATE = _TK_ATTRIBUTE
                attr = self._decode_attribute(row)
                if attr[0] in attribute_names:
                    raise BadAttributeName(attr[0], attribute_names[attr[0]])
                else:
                    attribute_names[attr[0]] = self._current_line
                obj['attributes'].append(attr)
                if isinstance(attr[1], (list, tuple)):
                    if encode_nominal:
                        conversor = EncodedNominalConversor(attr[1])
                    else:
                        conversor = NominalConversor(attr[1])
                else:
                    CONVERSOR_MAP = {'STRING': str, 'INTEGER': lambda x: int(float(x)), 'NUMERIC': float, 'REAL': float}
                    conversor = CONVERSOR_MAP[attr[1]]
                self._conversors.append(conversor)
            elif u_row.startswith(_TK_DATA):
                if STATE != _TK_ATTRIBUTE:
                    raise BadLayout()
                break
            elif u_row.startswith(_TK_COMMENT):
                pass
        else:
            raise BadLayout()

        def stream():
            for row in s:
                self._current_line += 1
                row = row.strip()
                if row and (not row.startswith(_TK_COMMENT)):
                    yield row
        obj['data'] = data.decode_rows(stream(), self._conversors)
        if obj['description'].endswith('\n'):
            obj['description'] = obj['description'][:-1]
        return obj

    def decode(self, s, encode_nominal=False, return_type=DENSE):
        """Returns the Python representation of a given ARFF file.

        When a file object is passed as an argument, this method reads lines
        iteratively, avoiding to load unnecessary information to the memory.

        :param s: a string or file object with the ARFF file.
        :param encode_nominal: boolean, if True perform a label encoding
            while reading the .arff file.
        :param return_type: determines the data structure used to store the
            dataset. Can be one of `arff.DENSE`, `arff.COO`, `arff.LOD`,
            `arff.DENSE_GEN` or `arff.LOD_GEN`.
            Consult the sections on `working with sparse data`_ and `loading
            progressively`_.
        """
        try:
            return self._decode(s, encode_nominal=encode_nominal, matrix_type=return_type)
        except ArffException as e:
            e.line = self._current_line
            raise e