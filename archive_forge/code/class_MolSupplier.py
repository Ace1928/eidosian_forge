class MolSupplier(object):
    """ we must, at minimum, support forward iteration

  """

    def __init__(self):
        raise ValueError('cannot instantiate MolSuppliers')

    def Reset(self):
        pass

    def __iter__(self):
        self.Reset()
        return self

    def next(self):
        res = self.NextMol()
        if res is not None:
            return res
        else:
            raise StopIteration

    def NextMol(self):
        """   Must be implemented in child class

    """
        pass
    __next__ = next