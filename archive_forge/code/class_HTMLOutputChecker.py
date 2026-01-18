import doctest
import xml.etree.ElementTree as ET
class HTMLOutputChecker(RealOutputChecker):

    def check_output(self, want, got, optionflags):
        normal = RealOutputChecker.check_output(self, want, got, optionflags)
        if normal or not got:
            return normal
        try:
            want_xml = make_xml(want)
        except XMLParseError:
            pass
        else:
            try:
                got_xml = make_xml(got)
            except XMLParseError:
                pass
            else:
                if xml_compare(want_xml, got_xml):
                    return True
        return False

    def output_difference(self, example, got, optionflags):
        actual = RealOutputChecker.output_difference(self, example, got, optionflags)
        want_xml = got_xml = None
        try:
            want_xml = make_xml(example.want)
            want_norm = make_string(want_xml)
        except XMLParseError as e:
            if example.want.startswith('<'):
                want_norm = '(bad XML: %s)' % e
            else:
                return actual
        try:
            got_xml = make_xml(got)
            got_norm = make_string(got_xml)
        except XMLParseError as e:
            if example.want.startswith('<'):
                got_norm = '(bad XML: %s)' % e
            else:
                return actual
        s = '%s\nXML Wanted: %s\nXML Got   : %s\n' % (actual, want_norm, got_norm)
        if got_xml and want_xml:
            result = []
            xml_compare(want_xml, got_xml, result.append)
            s += 'Difference report:\n%s\n' % '\n'.join(result)
        return s