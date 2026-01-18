from collections import defaultdict
class _HasMetadata(object):

    def set_metadata(self, name, node):
        """
        Attach unnamed metadata *node* to the metadata slot *name* of this
        value.
        """
        self.metadata[name] = node

    def _stringify_metadata(self, leading_comma=False):
        if self.metadata:
            buf = []
            if leading_comma:
                buf.append('')
            buf += ['!{0} {1}'.format(k, v.get_reference()) for k, v in self.metadata.items()]
            return ', '.join(buf)
        else:
            return ''