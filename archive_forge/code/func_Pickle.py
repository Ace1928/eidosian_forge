import pickle
def Pickle(self, fileName='foo.pkl'):
    """ Pickles the tree and writes it to disk

    """
    with open(fileName, 'wb+') as pFile:
        pickle.dump(self, pFile)