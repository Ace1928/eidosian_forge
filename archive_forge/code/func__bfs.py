import argparse
import logging
import os
import warnings
import numpy as np
import cv2
import mxnet as mx
def _bfs(root_node, process_node):
    """
    Implementation of Breadth-first search (BFS) on caffe network DAG
    :param root_node: root node of caffe network DAG
    :param process_node: function to run on each node
    """
    from collections import deque
    seen_nodes = set()
    next_nodes = deque()
    seen_nodes.add(root_node)
    next_nodes.append(root_node)
    while next_nodes:
        current_node = next_nodes.popleft()
        process_node(current_node)
        for child_node in current_node.children:
            if child_node not in seen_nodes:
                seen_nodes.add(child_node)
                next_nodes.append(child_node)