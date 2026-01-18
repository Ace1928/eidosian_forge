import pickle
def SaveState(self, fileName):
    """ Writes this calculator off to a file so that it can be easily loaded later

     **Arguments**

       - fileName: the name of the file to be written

    """
    try:
        f = open(fileName, 'wb+')
    except Exception:
        print('cannot open output file %s for writing' % fileName)
        return
    pickle.dump(self, f)
    f.close()