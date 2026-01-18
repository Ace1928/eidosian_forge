import unicodedata
def _member_list_(self):
    self._push('member_list')
    self._seq([lambda: self._bind(self._member_, 'm'), self._member_list__s1_, self._sp_, self._member_list__s3_, lambda: self._succeed([self._get('m')] + self._get('ms'))])
    self._pop('member_list')