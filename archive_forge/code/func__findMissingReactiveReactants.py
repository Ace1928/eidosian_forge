import itertools
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
from . import utils
def _findMissingReactiveReactants(rfps, pfps, currentReactants, unmappedPAtoms, output=False):
    if output:
        print('--- _findMissingReactiveReactants ---')
    if not len(unmappedPAtoms):
        return currentReactants
    else:
        finalReactants = []
        numReactants = len(rfps)
        for reacts, umPA in zip(currentReactants, unmappedPAtoms):
            finalReactants.append(reacts)
            if umPA[1] > 0:
                remainingReactants = set(range(numReactants)).difference(set(reacts))
                remainingReactants = sorted(remainingReactants, key=lambda x: rfps[x].reactivity / float(rfps[x].molecule.GetNumAtoms()), reverse=True)
                missingPAtoms = []
                for bit, c in umPA[-1]:
                    for pbi in range(len(pfps)):
                        if bit in pfps[pbi].bitInfoScaffoldFP:
                            a = pfps[pbi].bitInfoScaffoldFP[bit][0]
                            missingPAtoms.extend([pfps[pbi].molecule.GetAtomWithIdx(a[0]).GetAtomicNum()] * c)
                missingPAtoms = Counter(missingPAtoms)
                if output > 0:
                    print(missingPAtoms)
                queries = [(rdqueries.AtomNumEqualsQueryAtom(a), a) for a in missingPAtoms]
                maxFullfilledQueries = 0
                maxReactivity = -1
                addReactants = []
                for r in remainingReactants:
                    if output > 0:
                        print(' >> Reactant', r, rfps[r].reactivity / float(rfps[r].molecule.GetNumAtoms()))
                    countFullfilledQueries = 0
                    for q, a in queries:
                        if len(rfps[r].molecule.GetAtomsMatchingQuery(q)) >= missingPAtoms[a]:
                            countFullfilledQueries += 1
                    if output > 0:
                        print(' Max reactivity', maxReactivity)
                        print(' Max fulfilled queries', maxFullfilledQueries)
                    if countFullfilledQueries > maxFullfilledQueries:
                        maxFullfilledQueries = countFullfilledQueries
                        maxReactivity = rfps[r].reactivity / float(rfps[r].molecule.GetNumAtoms())
                        addReactants = [r]
                    elif maxFullfilledQueries and countFullfilledQueries == maxFullfilledQueries and (rfps[r].reactivity / float(rfps[r].molecule.GetNumAtoms()) >= maxReactivity):
                        maxFullfilledQueries = countFullfilledQueries
                        addReactants.append(r)
                    if output > 0:
                        print(' Added reactants', addReactants)
                finalReactants[-1].extend(addReactants)
    if output > 0:
        print(' >> Final reactants', finalReactants)
    return finalReactants