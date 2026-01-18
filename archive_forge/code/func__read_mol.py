import argparse
import logging
import sys
from rdkit import Chem
from rdkit.Chem.MolStandardize import Standardizer, Validator
def _read_mol(args):
    if args.smiles:
        return Chem.MolFromSmiles(args.smiles)
    elif args.intype in {'smi', 'smiles'} or args.infile.name.endswith('smi') or args.infile.name.endswith('smiles'):
        return Chem.MolFromSmiles(args.infile.read())
    elif args.intype in {'mol', 'sdf'} or args.infile.name.endswith('mol') or args.infile.name.endswith('sdf'):
        return Chem.MolFromMolBlock(args.infile.read())
    else:
        return Chem.MolFromSmiles(args.infile.read())