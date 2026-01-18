from __future__ import annotations
import collections
import logging
from typing import TYPE_CHECKING
import networkx as nx
import numpy as np
from monty.json import MSONable, jsanitize
from pymatgen.analysis.chemenv.connectivity.connected_components import ConnectedComponent
from pymatgen.analysis.chemenv.connectivity.environment_nodes import get_environment_node
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
def add_bonds(self, isite, site_neighbors_set):
    """
        Add the bonds for a given site index to the structure connectivity graph.

        Args:
            isite: Index of the site for which the bonds have to be added.
            site_neighbors_set: site_neighbors_set: Neighbors set of the site
        """
    existing_edges = self._graph.edges(nbunch=[isite], data=True)
    for nb_index_and_image in site_neighbors_set.neighb_indices_and_images:
        nb_index_unitcell = nb_index_and_image['index']
        nb_image_cell = nb_index_and_image['image_cell']
        exists = False
        if np.allclose(nb_image_cell, np.zeros(3)):
            for _, ineighb1, data1 in existing_edges:
                if np.allclose(data1['delta'], np.zeros(3)) and nb_index_unitcell == ineighb1:
                    exists = True
                    break
        elif isite == nb_index_unitcell:
            for isite1, ineighb1, data1 in existing_edges:
                if isite1 == ineighb1 and (np.allclose(data1['delta'], nb_image_cell) or np.allclose(data1['delta'], -nb_image_cell)):
                    exists = True
                    break
        else:
            for _, ineighb1, data1 in existing_edges:
                if nb_index_unitcell == ineighb1:
                    if data1['start'] == isite:
                        if np.allclose(data1['delta'], nb_image_cell):
                            exists = True
                            break
                    elif data1['end'] == isite:
                        if np.allclose(data1['delta'], -nb_image_cell):
                            exists = True
                            break
                    else:
                        raise ValueError('SHOULD NOT HAPPEN ???')
        if not exists:
            self._graph.add_edge(isite, nb_index_unitcell, start=isite, end=nb_index_unitcell, delta=nb_image_cell)