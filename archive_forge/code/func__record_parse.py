import codecs
import re
from io import StringIO
from xml.etree.ElementTree import Element, ElementTree, SubElement, TreeBuilder
from nltk.data import PathPointer, find
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