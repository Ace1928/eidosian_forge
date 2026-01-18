import itertools
import math
import operator
import random
from functools import reduce
class Sidechain:
    """Holds the name (identifier) and property list for the
    given sidechain/fragment.  Properties are assumed to
    be numerical values"""

    def __init__(self, name, props, goodCount=0, **extra_data):
        """name, props, goodCount=0 -> initialize a Sidechain
        initialize a sidechain.
        name: the unique name for the sidechain
        props: the property vector (see Properties class for details)
        goodCount: the number of times this reagent belongs to
            a good product, where good is a product that is in the desired
            property space.
        """
        self.name = name
        self.props = props
        self.good_count = goodCount
        self.dropped = False
        self.extra_data = extra_data

    def data(self):
        return self.extra_data

    def __str__(self):
        return 'Sidechain %s(%s, goodCount=%s, **%r)' % (self.name, self.props, self.good_count, self.extra_data)

    def __repr__(self):
        return 'Sidechain(%r, %r, %s, **%r)' % (self.name, self.props, self.good_count, self.extra_data)