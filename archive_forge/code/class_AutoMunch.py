from .python3_compat import iterkeys, iteritems, Mapping  #, u
class AutoMunch(Munch):

    def __setattr__(self, k, v):
        """ Works the same as Munch.__setattr__ but if you supply
            a dictionary as value it will convert it to another Munch.
        """
        if isinstance(v, Mapping) and (not isinstance(v, (AutoMunch, Munch))):
            v = munchify(v, AutoMunch)
        super(AutoMunch, self).__setattr__(k, v)