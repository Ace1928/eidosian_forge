import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.Polypeptide import PPBuilder
class FragmentMapper:
    """Map polypeptides in a model to lists of representative fragments."""

    def __init__(self, model, lsize=20, flength=5, fdir='.'):
        """Create instance of FragmentMapper.

        :param model: the model that will be mapped
        :type model: L{Model}

        :param lsize: number of fragments in the library
        :type lsize: int

        :param flength: length of fragments in the library
        :type flength: int

        :param fdir: directory where the definition files are
                     found (default=".")
        :type fdir: string
        """
        if flength == 5:
            self.edge = 2
        elif flength == 7:
            self.edge = 3
        else:
            raise PDBException('Fragment length should be 5 or 7.')
        self.flength = flength
        self.lsize = lsize
        self.reflist = _read_fragments(lsize, flength, fdir)
        self.model = model
        self.fd = self._map(self.model)

    def _map(self, model):
        """Map (PRIVATE).

        :param model: the model that will be mapped
        :type model: L{Model}
        """
        ppb = PPBuilder()
        ppl = ppb.build_peptides(model)
        fd = {}
        for pp in ppl:
            try:
                flist = _make_fragment_list(pp, self.flength)
                mflist = _map_fragment_list(flist, self.reflist)
                for i in range(len(pp)):
                    res = pp[i]
                    if i < self.edge:
                        continue
                    elif i >= len(pp) - self.edge:
                        continue
                    else:
                        index = i - self.edge
                        assert index >= 0
                        fd[res] = mflist[index]
            except PDBException as why:
                if why == 'CHAINBREAK':
                    pass
                else:
                    raise PDBException(why) from None
        return fd

    def __contains__(self, res):
        """Check if the given residue is in any of the mapped fragments.

        :type res: L{Residue}
        """
        return res in self.fd

    def __getitem__(self, res):
        """Get an entry.

        :type res: L{Residue}

        :return: fragment classification
        :rtype: L{Fragment}
        """
        return self.fd[res]