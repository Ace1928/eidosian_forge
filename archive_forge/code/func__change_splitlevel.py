from sqlparse import sql, tokens as T
def _change_splitlevel(self, ttype, value):
    """Get the new split level (increase, decrease or remain equal)"""
    if ttype not in T.Keyword:
        return 0
    unified = value.upper()
    if ttype is T.Keyword.DDL and unified.startswith('CREATE'):
        self._is_create = True
        return 0
    if unified == 'DECLARE' and self._is_create and (self._begin_depth == 0):
        self._in_declare = True
        return 1
    if unified == 'BEGIN':
        self._begin_depth += 1
        if self._is_create:
            return 1
        return 0
    if unified == 'END':
        self._begin_depth = max(0, self._begin_depth - 1)
        return -1
    if unified in ('IF', 'FOR', 'WHILE') and self._is_create and (self._begin_depth > 0):
        return 1
    if unified in ('END IF', 'END FOR', 'END WHILE'):
        return -1
    return 0