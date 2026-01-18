from reportlab import rl_config
class RLMonkeyPatchRandom(random.Random):

    def randrange(self, start, stop=None, step=1, _int=int, _maxwidth=1 << random.BPF):
        """Choose a random item from range(start, stop[, step]).

            This fixes the problem with randint() which includes the
            endpoint; in Python this is usually not what you want.

            """
        istart = _int(start)
        if istart != start:
            raise ValueError('non-integer arg 1 for randrange()')
        if stop is None:
            if istart > 0:
                if istart >= _maxwidth:
                    return self._randbelow(istart)
                return _int(self.random() * istart)
            raise ValueError('empty range for randrange()')
        istop = _int(stop)
        if istop != stop:
            raise ValueError('non-integer stop for randrange()')
        width = istop - istart
        if step == 1 and width > 0:
            if width >= _maxwidth:
                return _int(istart + self._randbelow(width))
            return _int(istart + _int(self.random() * width))
        if step == 1:
            raise ValueError('empty range for randrange() (%d,%d, %d)' % (istart, istop, width))
        istep = _int(step)
        if istep != step:
            raise ValueError('non-integer step for randrange()')
        if istep > 0:
            n = (width + istep - 1) // istep
        elif istep < 0:
            n = (width + istep + 1) // istep
        else:
            raise ValueError('zero step for randrange()')
        if n <= 0:
            raise ValueError('empty range for randrange()')
        if n >= _maxwidth:
            return istart + istep * self._randbelow(n)
        return istart + istep * _int(self.random() * n)

    def choice(self, seq):
        """Choose a random element from a non-empty sequence."""
        return seq[int(self.random() * len(seq))]