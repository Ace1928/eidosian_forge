from fontTools.encodings.StandardEncoding import StandardEncoding
class ps_font(ps_object):

    def __str__(self):
        psstring = '%d dict dup begin\n' % len(self.value)
        for key in _type1_pre_eexec_order:
            try:
                value = self.value[key]
            except KeyError:
                pass
            else:
                psstring = psstring + _type1_item_repr(key, value)
        items = sorted(self.value.items())
        for key, value in items:
            if key not in _type1_pre_eexec_order + _type1_post_eexec_order:
                psstring = psstring + _type1_item_repr(key, value)
        psstring = psstring + 'currentdict end\ncurrentfile eexec\ndup '
        for key in _type1_post_eexec_order:
            try:
                value = self.value[key]
            except KeyError:
                pass
            else:
                psstring = psstring + _type1_item_repr(key, value)
        return psstring + 'dup/FontName get exch definefont pop\nmark currentfile closefile\n' + 8 * (64 * '0' + '\n') + 'cleartomark' + '\n'

    def __repr__(self):
        return '<font>'