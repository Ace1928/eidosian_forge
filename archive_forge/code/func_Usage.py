import numpy
from PIL import Image, ImageDraw
def Usage():
    """ provides a list of arguments for when this is used from the command line

  """
    import sys
    print('Usage: VoteImg.py [optional arguments] <modelfile.pkl> <datafile.qdat>')
    print('Optional Arguments:')
    print('\t-o outfilename: the name of the output image file.')
    print('\t                The extension determines the type of image saved.')
    print('\t-b: only include bad (misclassified) examples')
    print('\t-t: sort the results by the true (input) classification')
    print('\t-x scale: scale the image along the x axis (default: 10)')
    print('\t-y scale: scale the image along the y axis (default: 2)')
    print('\t-d databasename: instead of using a qdat file, pull the data from')
    print('\t                 a database.  In this case the filename argument')
    print('\t                 is used to indicate the name of the table in the database.')
    sys.exit(-1)