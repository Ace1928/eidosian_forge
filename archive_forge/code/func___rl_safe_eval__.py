import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_safe_eval__(self, expr, g, l, mode, timeout=None, allowed_magic_methods=None, __frame_depth__=3):
    bcode, ns = self.__rl_compile__(expr, fname='<string>', mode=mode, flags=0, inherit=True, visit=UntrustedAstTransformer(nameIsAllowed=self.__rl_is_allowed_name__).visit)
    if None in (l, g):
        G = sys._getframe(__frame_depth__)
        L = G.f_locals.copy() if l is None else l
        G = G.f_globals.copy() if g is None else g
    else:
        G = g
        L = l
    obi = (G['__builtins__'],) if '__builtins__' in G else False
    G['__builtins__'] = self.__rl_builtins__
    self.__rl_limit__ = self.__time_time__() + (timeout if timeout is not None else self.timeout)
    if allowed_magic_methods is not None:
        self.allowed_magic_methods = (__allowed_magic_methods__ if allowed_magic_methods == True else allowed_magic_methods) if allowed_magic_methods else []
    sbi = [].append
    bi = self.real_bi
    bir = self.bi_replace
    for n, r in bir:
        sbi(getattr(bi, n))
        setattr(bi, n, r)
    try:
        return eval(bcode, G, L)
    finally:
        sbi = sbi.__self__
        for i, (n, r) in enumerate(bir):
            setattr(bi, n, sbi[i])
        if obi:
            G['__builtins__'] = obi[0]