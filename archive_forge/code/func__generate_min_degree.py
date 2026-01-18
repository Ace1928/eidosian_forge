import itertools
import math
import networkx as nx
from networkx.utils import py_random_state
def _generate_min_degree(gamma, average_degree, max_degree, tolerance, max_iters):
    """Returns a minimum degree from the given average degree."""
    try:
        from scipy.special import zeta
    except ImportError:

        def zeta(x, q):
            return _hurwitz_zeta(x, q, tolerance)
    min_deg_top = max_degree
    min_deg_bot = 1
    min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
    itrs = 0
    mid_avg_deg = 0
    while abs(mid_avg_deg - average_degree) > tolerance:
        if itrs > max_iters:
            raise nx.ExceededMaxIterations('Could not match average_degree')
        mid_avg_deg = 0
        for x in range(int(min_deg_mid), max_degree + 1):
            mid_avg_deg += x ** (-gamma + 1) / zeta(gamma, min_deg_mid)
        if mid_avg_deg > average_degree:
            min_deg_top = min_deg_mid
            min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
        else:
            min_deg_bot = min_deg_mid
            min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
        itrs += 1
    return round(min_deg_mid)