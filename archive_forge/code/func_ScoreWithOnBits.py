def ScoreWithOnBits(self, other):
    """ other must support GetOnBits() """
    obl = other.GetOnBits()
    cnt = 0
    for bit in self.GetBits():
        if bit in obl:
            cnt += 1
    return cnt