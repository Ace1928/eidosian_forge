import copy
import re
import sys
from rdkit import Chem
from rdkit import RDRandom as random
from rdkit.Chem import rdChemReactions as Reactions
def BRICSBuild(fragments, onlyCompleteMols=True, seeds=None, uniquify=True, scrambleReagents=True, maxDepth=3):
    seen = set()
    if not seeds:
        seeds = list(fragments)
    if scrambleReagents:
        seeds = list(seeds)
        random.shuffle(seeds, random=random.random)
    if scrambleReagents:
        tempReactions = list(reverseReactions)
        random.shuffle(tempReactions, random=random.random)
    else:
        tempReactions = reverseReactions
    for seed in seeds:
        seedIsR1 = False
        seedIsR2 = False
        nextSteps = []
        for rxn in tempReactions:
            if seed.HasSubstructMatch(rxn._matchers[0]):
                seedIsR1 = True
            if seed.HasSubstructMatch(rxn._matchers[1]):
                seedIsR2 = True
            for fragment in fragments:
                ps = None
                if fragment.HasSubstructMatch(rxn._matchers[0]):
                    if seedIsR2:
                        ps = rxn.RunReactants((fragment, seed))
                if fragment.HasSubstructMatch(rxn._matchers[1]):
                    if seedIsR1:
                        ps = rxn.RunReactants((seed, fragment))
                if ps:
                    for p in ps:
                        if uniquify:
                            pSmi = Chem.MolToSmiles(p[0], True)
                            if pSmi in seen:
                                continue
                            else:
                                seen.add(pSmi)
                        if p[0].HasSubstructMatch(dummyPattern):
                            nextSteps.append(p[0])
                            if not onlyCompleteMols:
                                yield p[0]
                        else:
                            yield p[0]
        if nextSteps and maxDepth > 0:
            for p in BRICSBuild(fragments, onlyCompleteMols=onlyCompleteMols, seeds=nextSteps, uniquify=uniquify, maxDepth=maxDepth - 1, scrambleReagents=scrambleReagents):
                if uniquify:
                    pSmi = Chem.MolToSmiles(p, True)
                    if pSmi in seen:
                        continue
                    else:
                        seen.add(pSmi)
                yield p