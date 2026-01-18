from ..util import FeedParserDict
def _start_admin_errorreportsto(self, attrs_d):
    self.push('errorreportsto', 1)
    value = self._get_attribute(attrs_d, 'rdf:resource')
    if value:
        self.elementstack[-1][2].append(value)
    self.pop('errorreportsto')