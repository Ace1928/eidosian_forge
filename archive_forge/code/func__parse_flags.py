from ._constants import *
def _parse_flags(source, state, char):
    sourceget = source.get
    add_flags = 0
    del_flags = 0
    if char != '-':
        while True:
            flag = FLAGS[char]
            if source.istext:
                if char == 'L':
                    msg = "bad inline flags: cannot use 'L' flag with a str pattern"
                    raise source.error(msg)
            elif char == 'u':
                msg = "bad inline flags: cannot use 'u' flag with a bytes pattern"
                raise source.error(msg)
            add_flags |= flag
            if flag & TYPE_FLAGS and add_flags & TYPE_FLAGS != flag:
                msg = "bad inline flags: flags 'a', 'u' and 'L' are incompatible"
                raise source.error(msg)
            char = sourceget()
            if char is None:
                raise source.error('missing -, : or )')
            if char in ')-:':
                break
            if char not in FLAGS:
                msg = 'unknown flag' if char.isalpha() else 'missing -, : or )'
                raise source.error(msg, len(char))
    if char == ')':
        state.flags |= add_flags
        return None
    if add_flags & GLOBAL_FLAGS:
        raise source.error('bad inline flags: cannot turn on global flag', 1)
    if char == '-':
        char = sourceget()
        if char is None:
            raise source.error('missing flag')
        if char not in FLAGS:
            msg = 'unknown flag' if char.isalpha() else 'missing flag'
            raise source.error(msg, len(char))
        while True:
            flag = FLAGS[char]
            if flag & TYPE_FLAGS:
                msg = "bad inline flags: cannot turn off flags 'a', 'u' and 'L'"
                raise source.error(msg)
            del_flags |= flag
            char = sourceget()
            if char is None:
                raise source.error('missing :')
            if char == ':':
                break
            if char not in FLAGS:
                msg = 'unknown flag' if char.isalpha() else 'missing :'
                raise source.error(msg, len(char))
    assert char == ':'
    if del_flags & GLOBAL_FLAGS:
        raise source.error('bad inline flags: cannot turn off global flag', 1)
    if add_flags & del_flags:
        raise source.error('bad inline flags: flag turned on and off', 1)
    return (add_flags, del_flags)