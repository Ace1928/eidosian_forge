import os
from collections import defaultdict
import numpy as np
import minerl
import time
from typing import List
import collections
import os
import cv2
import gym
from minerl.data import DataPipeline
import sys
import time
from collections import deque, defaultdict
from enum import Enum
class VisTools:
    """
    number of methods to draw chains with pyGraphviz
    """

    @staticmethod
    def get_all_vertexes_from_edges(edges):
        """
        determines all vertex of a graph
        :param edges: list of edges
        :return: list of vertexes
        """
        vertexes = []
        for left, right in edges:
            if left not in vertexes:
                vertexes.append(left)
            if right not in vertexes:
                vertexes.append(right)
        return vertexes

    @staticmethod
    def get_colored_vertexes(chain):
        """
        determines the color for each vertex
        item + its actions have the same color
        :param chain:
        :return: {vertex: color}
        """
        vertexes = VisTools.get_all_vertexes_from_edges(chain)
        result = {}
        colors = ['#ffe6cc', '#ccffe6']
        current_color = 0
        for vertex in vertexes:
            result[vertex] = colors[current_color]
            bool_ = True
            for action in ['equip', 'craft', 'nearbyCraft', 'nearbySmelt', 'place']:
                if action + ':' in vertex:
                    bool_ = False
            if bool_:
                current_color = (current_color + 1) % len(colors)
        return result

    @staticmethod
    def replace_with_name(name):
        """
        replace all names with human readable variants
        :param name: crafting action or item (item string will be skipped)
        :return: human readable name
        """
        if len(name.split(':')) == 3:
            name, order, digit = name.split(':')
            name = name + ':' + order
            translate = {'place': ['cobblestone', 'crafting_table', 'dirt', 'furnace', 'none', 'stone', 'torch'], 'nearbySmelt': ['coal', 'iron_ingot', 'none'], 'nearbyCraft': ['furnace', 'iron_axe', 'iron_pickaxe', 'none', 'stone_axe', 'stone_pickaxe', 'wooden_axe', 'wooden_pickaxe'], 'equip': ['air', 'iron_axe', 'iron_pickaxe', 'none', 'stone_axe', 'stone_pickaxe', 'wooden_axe', 'wooden_pickaxe'], 'craft': ['crafting_table', 'none', 'planks', 'stick', 'torch']}
            name_without_digits = name
            while name_without_digits not in translate:
                name_without_digits = name_without_digits[:-1]
            return name + ' -> ' + translate[name_without_digits][int(digit)]
        else:
            return name

    @staticmethod
    def draw_graph(file_name, graph, format_='svg', vertex_colors=None):
        """
        drawing png graph from the list of edges
        :param vertex_colors:
        :param format_: resulted file format
        :param file_name: file_name
        :param graph: graph file with format: (left_edge, right_edge) or (left_edge, right_edge, label)
        :return: None
        """
        import pygraphviz as pgv
        g_out = pgv.AGraph(strict=False, directed=True)
        for i in graph:
            g_out.add_edge(i[0], i[1], color='black')
            edge = g_out.get_edge(i[0], i[1])
            if len(i) > 2:
                edge.attr['label'] = i[2]
        g_out.node_attr['style'] = 'filled'
        if vertex_colors:
            for vertex, color in vertex_colors.items():
                g_out.get_node(vertex).attr['fillcolor'] = color
        g_out.layout(prog='dot')
        g_out.draw(path='{file_name}.{format_}'.format(**locals()))

    @staticmethod
    def save_chain_in_graph(chain_to_save, name='out', format_='png'):
        """
        saving image of a graph using draw_graph method
        :param chain_to_save:
        :param name: filename
        :param format_: file type e.g. ".png" or ".svg"
        :return:
        """
        graph = []
        for c_index, item in enumerate(chain_to_save):
            if c_index:
                graph.append([str(c_index) + '\n' + VisTools.replace_with_name(chain_to_save[c_index - 1]), str(c_index + 1) + '\n' + VisTools.replace_with_name(item)])
        VisTools.draw_graph(name, graph=graph, format_=format_, vertex_colors=VisTools.get_colored_vertexes(graph))