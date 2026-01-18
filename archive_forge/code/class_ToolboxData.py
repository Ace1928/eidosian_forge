import codecs
import re
from io import StringIO
from xml.etree.ElementTree import Element, ElementTree, SubElement, TreeBuilder
from nltk.data import PathPointer, find
class ToolboxData(StandardFormat):

    def parse(self, grammar=None, **kwargs):
        if grammar:
            return self._chunk_parse(grammar=grammar, **kwargs)
        else:
            return self._record_parse(**kwargs)

    def _record_parse(self, key=None, **kwargs):
        """
        Returns an element tree structure corresponding to a toolbox data file with
        all markers at the same level.

        Thus the following Toolbox database::
            \\_sh v3.0  400  Rotokas Dictionary
            \\_DateStampHasFourDigitYear

            \\lx kaa
            \\ps V.A
            \\ge gag
            \\gp nek i pas

            \\lx kaa
            \\ps V.B
            \\ge strangle
            \\gp pasim nek

        after parsing will end up with the same structure (ignoring the extra
        whitespace) as the following XML fragment after being parsed by
        ElementTree::
            <toolbox_data>
                <header>
                    <_sh>v3.0  400  Rotokas Dictionary</_sh>
                    <_DateStampHasFourDigitYear/>
                </header>

                <record>
                    <lx>kaa</lx>
                    <ps>V.A</ps>
                    <ge>gag</ge>
                    <gp>nek i pas</gp>
                </record>

                <record>
                    <lx>kaa</lx>
                    <ps>V.B</ps>
                    <ge>strangle</ge>
                    <gp>pasim nek</gp>
                </record>
            </toolbox_data>

        :param key: Name of key marker at the start of each record. If set to
            None (the default value) the first marker that doesn't begin with
            an underscore is assumed to be the key.
        :type key: str
        :param kwargs: Keyword arguments passed to ``StandardFormat.fields()``
        :type kwargs: dict
        :rtype: ElementTree._ElementInterface
        :return: contents of toolbox data divided into header and records
        """
        builder = TreeBuilder()
        builder.start('toolbox_data', {})
        builder.start('header', {})
        in_records = False
        for mkr, value in self.fields(**kwargs):
            if key is None and (not in_records) and (mkr[0] != '_'):
                key = mkr
            if mkr == key:
                if in_records:
                    builder.end('record')
                else:
                    builder.end('header')
                    in_records = True
                builder.start('record', {})
            builder.start(mkr, {})
            builder.data(value)
            builder.end(mkr)
        if in_records:
            builder.end('record')
        else:
            builder.end('header')
        builder.end('toolbox_data')
        return builder.close()

    def _tree2etree(self, parent):
        from nltk.tree import Tree
        root = Element(parent.label())
        for child in parent:
            if isinstance(child, Tree):
                root.append(self._tree2etree(child))
            else:
                text, tag = child
                e = SubElement(root, tag)
                e.text = text
        return root

    def _chunk_parse(self, grammar=None, root_label='record', trace=0, **kwargs):
        """
        Returns an element tree structure corresponding to a toolbox data file
        parsed according to the chunk grammar.

        :type grammar: str
        :param grammar: Contains the chunking rules used to parse the
            database.  See ``chunk.RegExp`` for documentation.
        :type root_label: str
        :param root_label: The node value that should be used for the
            top node of the chunk structure.
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            ``1`` will generate normal tracing output; and ``2`` or
            higher will generate verbose tracing output.
        :type kwargs: dict
        :param kwargs: Keyword arguments passed to ``toolbox.StandardFormat.fields()``
        :rtype: ElementTree._ElementInterface
        """
        from nltk import chunk
        from nltk.tree import Tree
        cp = chunk.RegexpParser(grammar, root_label=root_label, trace=trace)
        db = self.parse(**kwargs)
        tb_etree = Element('toolbox_data')
        header = db.find('header')
        tb_etree.append(header)
        for record in db.findall('record'):
            parsed = cp.parse([(elem.text, elem.tag) for elem in record])
            tb_etree.append(self._tree2etree(parsed))
        return tb_etree