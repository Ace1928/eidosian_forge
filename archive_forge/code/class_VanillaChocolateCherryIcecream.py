from __future__ import print_function
from Universe import Icecream, Truck
class VanillaChocolateCherryIcecream(VanillaChocolateIcecream):

    def __init__(self, flavor=''):
        super(VanillaChocolateIcecream, self).__init__(flavor)

    def clone(self):
        return VanillaChocolateCherryIcecream(self.getFlavor())

    def getFlavor(self):
        base_flavor = super(VanillaChocolateCherryIcecream, self).getFlavor()
        return base_flavor + ' and a cherry'