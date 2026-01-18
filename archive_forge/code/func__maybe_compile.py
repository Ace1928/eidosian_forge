import __future__
import warnings
def _maybe_compile(compiler, source, filename, symbol):
    for line in source.split('\n'):
        line = line.strip()
        if line and line[0] != '#':
            break
    else:
        if symbol != 'eval':
            source = 'pass'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', (SyntaxWarning, DeprecationWarning))
        try:
            compiler(source, filename, symbol)
        except SyntaxError:
            try:
                compiler(source + '\n', filename, symbol)
                return None
            except SyntaxError as e:
                if 'incomplete input' in str(e):
                    return None
    return compiler(source, filename, symbol, incomplete_input=False)