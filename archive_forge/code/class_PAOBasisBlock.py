from ase.calculators.calculator import Parameters
class PAOBasisBlock(Parameters):
    """
    Representing a block in PAO.Basis for one species.
    """

    def __init__(self, block):
        """
        Parameters:
            -block : String. A block defining the basis set of a single
                     species using the format of a PAO.Basis block.
                     The initial label should be left out since it is
                     determined programatically.
                     Example1: 2 nodes 1.0
                               n=2 0 2 E 50.0 2.5
                               3.50 3.50
                               0.95 1.00
                               1 1 P 2
                               3.50
                     Example2: 1
                               0 2 S 0.2
                               5.00 0.00
                     See siesta manual for details.
        """
        assert isinstance(block, str)
        Parameters.__init__(self, block=block)

    def script(self, label):
        """
        Write the fdf script for the block.

        Parameters:
            -label : The label to insert in front of the block.
        """
        return label + ' ' + self['block']