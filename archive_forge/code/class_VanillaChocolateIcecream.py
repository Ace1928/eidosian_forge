from __future__ import print_function
from Universe import Icecream, Truck
class VanillaChocolateIcecream(Icecream):

    def __init__(self, flavor=''):
        super(VanillaChocolateIcecream, self).__init__(flavor)

    def clone(self):
        return VanillaChocolateIcecream(self.getFlavor())

    def getFlavor(self):
        return 'vanilla sprinked with chocolate'