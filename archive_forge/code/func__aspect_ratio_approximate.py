from decimal import Decimal
from kivy.lang import Builder
from kivy.properties import ListProperty
def _aspect_ratio_approximate(self, size):
    return Decimal('%.2f' % (float(size[0]) / size[1]))