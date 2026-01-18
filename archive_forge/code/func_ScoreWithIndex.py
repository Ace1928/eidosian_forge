def ScoreWithIndex(self, other):
    """ other must support __getitem__() """
    cnt = 0
    for bit in self.GetBits():
        if other[bit]:
            cnt += 1
    return cnt