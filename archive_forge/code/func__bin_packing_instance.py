from pulp import *
import random
from itertools import product
def _bin_packing_instance(bins, seed=0):
    packed_bins = [[] for _ in range(bins)]
    bin_size = bins * 100
    random.seed(seed)
    for i in range(len(packed_bins)):
        remaining_size = bin_size
        while remaining_size >= 1:
            item = random.randrange(1, remaining_size + 10)
            packed_bins[i].append(item)
            remaining_size -= item
        packed_bins[i][-1] += remaining_size
    all_items_with_bin = [(n, i) for i, l in enumerate(packed_bins) for n in l]
    random.shuffle(all_items_with_bin)
    items, packing = zip(*all_items_with_bin)
    return (items, packing, bin_size)