import os
import re
from rdkit import RDConfig
def SplitComposition(compStr):
    """ Takes a simple chemical composition and turns into a list of element,# pairs.

        i.e. 'Fe3Al' -> [('Fe',3),('Al',1)]

        **Arguments**

         - compStr: the composition string to be processed

        **Returns**

         - the *composVect* corresponding to _compStr_

        **Note**

          -this isn't smart enough by half to deal with anything even
              remotely subtle, so be gentle.

    """
    target = '([A-Z][a-z]?)([0-9\\.]*)'
    theExpr = re.compile(target)
    matches = theExpr.findall(compStr)
    res = []
    for match in matches:
        if len(match[1]) > 0:
            res.append((match[0], float(match[1])))
        else:
            res.append((match[0], 1))
    return res