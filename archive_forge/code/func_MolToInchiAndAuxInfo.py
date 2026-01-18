import logging
from rdkit import RDLogger
from rdkit.Chem import rdinchi
def MolToInchiAndAuxInfo(mol, options='', logLevel=None, treatWarningAsError=False):
    """Returns the standard InChI string and InChI auxInfo for a molecule

    Keyword arguments:
    logLevel -- the log level used for logging logs and messages from InChI
    API. set to None to diable the logging completely
    treatWarningAsError -- set to True to raise an exception in case of a
    molecule that generates warning in calling InChI API. The resultant InChI
    string and AuxInfo string as well as the error message are encoded in the
    exception.

    Returns:
    a tuple of the standard InChI string and the auxInfo string returned by
    InChI API, in that order, for the input molecule
    """
    inchi, retcode, message, logs, aux = rdinchi.MolToInchi(mol, options)
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
        raise InchiReadWriteError(inchi, aux, message)
    return (inchi, aux)