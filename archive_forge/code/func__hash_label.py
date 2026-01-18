from collections import Counter, defaultdict
from hashlib import blake2b
import networkx as nx
def _hash_label(label, digest_size):
    return blake2b(label.encode('ascii'), digest_size=digest_size).hexdigest()