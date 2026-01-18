import sys
import glob
import inspect
def P(obj, m=mod.__name__, CT=type):
    return type(obj) == CT and obj.__module__ == m