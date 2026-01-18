import logging
from rdkit import RDLogger
from rdkit.Chem import rdinchi
def MolFromInchi(inchi, sanitize=True, removeHs=True, logLevel=None, treatWarningAsError=False):
    """Construct a molecule from a InChI string

    Keyword arguments:
    sanitize -- set to True to enable sanitization of the molecule. Default is
    True
    removeHs -- set to True to remove Hydrogens from a molecule. This only
    makes sense when sanitization is enabled
    logLevel -- the log level used for logging logs and messages from InChI
    API. set to None to diable the logging completely
    treatWarningAsError -- set to True to raise an exception in case of a
    molecule that generates warning in calling InChI API. The resultant
    molecule  and error message are part of the excpetion

    Returns:
    a rdkit.Chem.rdchem.Mol instance
    """
    try:
        mol, retcode, message, log = rdinchi.InchiToMol(inchi, sanitize, removeHs)
    except ValueError as e:
        logger.error(str(e))
        return None
    if logLevel is not None:
        if logLevel not in logLevelToLogFunctionLookup:
            raise ValueError('Unsupported log level: %d' % logLevel)
        log = logLevelToLogFunctionLookup[logLevel]
        if retcode == 0:
            log(message)
    if retcode != 0:
        if retcode == 1:
            logger.warning(message)
        else:
            logger.error(message)
    if treatWarningAsError and retcode != 0:
        raise InchiReadWriteError(mol, message)
    return mol