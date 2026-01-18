import collections
import json
import sys
def ParseTarget(target):
    target, _, suffix = target.partition('#')
    filename, _, target = target.partition(':')
    return (filename, target, suffix)