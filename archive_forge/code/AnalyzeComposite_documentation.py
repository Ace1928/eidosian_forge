from warnings import warn
import pickle
import sys
import numpy
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import ScreenComposite
from rdkit.ML.Data import Stats
from rdkit.ML.DecTree import Tree, TreeUtils
 command line utility to report on the contributions of descriptors to
tree-based composite models

Usage:  AnalyzeComposite [optional args] <models>

      <models>: file name(s) of pickled composite model(s)
        (this is the name of the db table if using a database)

    Optional Arguments:

      -n number: the number of levels of each model to consider

      -d dbname: the database from which to read the models

      -N Note: the note string to search for to pull models from the database

      -v: be verbose whilst screening
