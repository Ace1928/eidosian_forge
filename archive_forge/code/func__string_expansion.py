import sys
def _string_expansion(self, name, value, explode, prefix):
    if value is None:
        return None
    tuples, items = is_list_of_tuples(value)
    if list_test(value) and (not tuples):
        return ','.join((quote(v, self.safe) for v in value))
    if dict_test(value) or tuples:
        items = items or sorted(value.items())
        format_str = '%s=%s' if explode else '%s,%s'
        return ','.join((format_str % (quote(k, self.safe), quote(v, self.safe)) for k, v in items))
    value = value[:prefix] if prefix else value
    return quote(value, self.safe)