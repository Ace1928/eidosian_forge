from collections import defaultdict
def _stringify_metadata(self, leading_comma=False):
    if self.metadata:
        buf = []
        if leading_comma:
            buf.append('')
        buf += ['!{0} {1}'.format(k, v.get_reference()) for k, v in self.metadata.items()]
        return ', '.join(buf)
    else:
        return ''