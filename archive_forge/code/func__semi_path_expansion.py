import sys
def _semi_path_expansion(self, name, value, explode, prefix):
    """Expansion method for ';' operator."""
    join_str = self.join_str
    safe = self.safe
    if value is None:
        return None
    if self.operator == '?':
        join_str = '&'
    tuples, items = is_list_of_tuples(value)
    if list_test(value) and (not tuples):
        if explode:
            expanded = join_str.join(('{}={}'.format(name, quote(v, safe)) for v in value if v is not None))
            return expanded if expanded else None
        else:
            value = ','.join((quote(v, safe) for v in value))
            return '{}={}'.format(name, value)
    if dict_test(value) or tuples:
        items = items or sorted(value.items())
        if explode:
            return join_str.join(('{}={}'.format(quote(k, safe), quote(v, safe)) for k, v in items if v is not None))
        else:
            expanded = ','.join(('{},{}'.format(quote(k, safe), quote(v, safe)) for k, v in items if v is not None))
            return '{}={}'.format(name, expanded)
    value = value[:prefix] if prefix else value
    if value:
        return '{}={}'.format(name, quote(value, safe))
    return name