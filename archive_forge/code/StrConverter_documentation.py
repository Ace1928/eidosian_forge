import numpy as np
import matplotlib.units as units

    A Matplotlib converter class for string data values.

    Valid units for string are:
    - 'indexed' : Values are indexed as they are specified for plotting.
    - 'sorted'  : Values are sorted alphanumerically.
    - 'inverted' : Values are inverted so that the first value is on top.
    - 'sorted-inverted' :  A combination of 'sorted' and 'inverted'
    