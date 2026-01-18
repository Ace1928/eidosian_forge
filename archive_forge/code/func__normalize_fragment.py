from warnings import warn
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from .utils import memoized_property
def _normalize_fragment(self, mol):
    for _ in range(self.max_restarts):
        for normalization in self.normalizations:
            product = self._apply_transform(mol, normalization.transform)
            if product:
                log.info(f'Rule applied: {normalization.name}')
                mol = product
                break
        else:
            return mol
    log.warning(f'Gave up normalization after {self.max_restarts} restarts')
    return mol