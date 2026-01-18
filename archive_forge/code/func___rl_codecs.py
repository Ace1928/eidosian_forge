from collections import namedtuple
import codecs
@staticmethod
def __rl_codecs(name, cache=__rl_codecs_cache, data=__rl_codecs_data, extension_codecs=__rl_extension_codecs, _256=True):
    try:
        return cache[name]
    except KeyError:
        if name in extension_codecs:
            x = extension_codecs[name]
            e, r = data[x.baseName]
            if x.exceptions:
                if e:
                    e = e.copy()
                    e.update(x.exceptions)
                else:
                    e = x.exceptions
            if x.rexceptions:
                if r:
                    r = r.copy()
                    r.update(x.rexceptions)
                else:
                    r = x.exceptions
        else:
            e, r = data[name]
        cache[name] = c = RL_Codecs._256_exception_codec(name, e, r) if _256 else RL_Codecs._makeCodecInfo(name, e, r or {})
    return c