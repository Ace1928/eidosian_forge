from operator import inv
def _make_opp_dict():

    def swap(t):
        return (t[1], t[0])
    dic = {(0, 1): (2, 3), (2, 0): (1, 3), (1, 2): (0, 3)}
    for k in list(dic):
        dic[dic[k]] = k
        dic[swap(dic[k])] = swap(k)
        dic[swap(k)] = swap(dic[k])
    return dic