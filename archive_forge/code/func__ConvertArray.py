def _ConvertArray(arr):
    return ', '.join(map(lambda x: '"%s"' % x, arr))