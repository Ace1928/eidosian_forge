from logging import getLogger
from pyomo.common.dependencies import attempt_import
from pyomo.core import (
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.expr.visitor import replace_expressions, identify_variables
from pyomo.contrib.community_detection.community_graph import generate_model_graph
from pyomo.common.dependencies import networkx as nx
from pyomo.common.dependencies.matplotlib import pyplot as plt
from itertools import combinations
import copy
class CommunityMap(object):
    """
    This class is used to create CommunityMap objects which are returned by the detect_communities function. Instances
    of this class allow dict-like usage and store relevant information about the given community map, such as the
    model used to create them, their networkX representation, etc.

    The CommunityMap object acts as a Python dictionary, mapping integer keys to tuples containing two lists
    (which contain the components in the given community) - a constraint list and variable list.

    Methods:
    generate_structured_model
    visualize_model_graph
    """

    def __init__(self, community_map, type_of_community_map, with_objective, weighted_graph, random_seed, use_only_active_components, model, graph, graph_node_mapping, constraint_variable_map, graph_partition):
        """
        Constructor method for the CommunityMap class

        Parameters
        ----------
        community_map: dict
            a Python dictionary that maps arbitrary keys (in this case, integers from zero to the number of
            communities minus one) to two-list tuples containing Pyomo components in the given community
        type_of_community_map: str
            a string that specifies the type of community map to be returned, the default is 'constraint'.
            'constraint' returns a dictionary (community_map) with communities based on constraint nodes,
            'variable' returns a dictionary (community_map) with communities based on variable nodes,
            'bipartite' returns a dictionary (community_map) with communities based on a bipartite graph (both constraint
            and variable nodes)
        with_objective: bool
            a Boolean argument that specifies whether or not the objective function is
            included in the model graph (and thus in 'community_map'); the default is True
        weighted_graph: bool
            a Boolean argument that specifies whether community_map is created based on a weighted model graph or an
            unweighted model graph; the default is True (type_of_community_map='bipartite' creates an unweighted
            model graph regardless of this parameter)
        random_seed: int or None
            an integer that is used as the random seed for the (heuristic) Louvain community detection
        use_only_active_components: bool, optional
            a Boolean argument that specifies whether inactive constraints/objectives are included in the community map
        model: Block
            a Pyomo model or block to be used for community detection
        graph: nx.Graph
            a NetworkX graph with nodes and edges based on the Pyomo optimization model
        graph_node_mapping: dict
            a dictionary that maps a number (which corresponds to a node in the networkX graph representation of the
            model) to a component in the model
        constraint_variable_map: dict
            a dictionary that maps a numbered constraint to a list of (numbered) variables that appear in the constraint
        graph_partition: dict
            the partition of the networkX model graph based on the Louvain community detection
        """
        self.community_map = community_map
        self.type_of_community_map = type_of_community_map
        self.with_objective = with_objective
        self.weighted_graph = weighted_graph
        self.random_seed = random_seed
        self.use_only_active_components = use_only_active_components
        self.model = model
        self.graph = graph
        self.graph_node_mapping = graph_node_mapping
        self.constraint_variable_map = constraint_variable_map
        self.graph_partition = graph_partition

    def __repr__(self):
        """

        repr method changed to return the community_map with the memory locations of the Pyomo components - use str
        method if the strings of the components are desired

        """
        return str(self.community_map)

    def __str__(self):
        """

        str method changed to return the community_map with the strings of the Pyomo components (user-friendly output)

        """
        str_community_map = dict()
        for key in self.community_map:
            str_community_map[key] = ([str(component) for component in self.community_map[key][0]], [str(component) for component in self.community_map[key][1]])
        return str(str_community_map)

    def __eq__(self, other):
        if isinstance(other, dict):
            return self.community_map == other
        elif isinstance(other, CommunityMap):
            return self.community_map == other.community_map
        return False

    def __iter__(self):
        for key in self.community_map:
            yield key

    def __getitem__(self, item):
        return self.community_map[item]

    def __len__(self):
        return len(self.community_map)

    def keys(self):
        return self.community_map.keys()

    def values(self):
        return self.community_map.values()

    def items(self):
        return self.community_map.items()

    def visualize_model_graph(self, type_of_graph='constraint', filename=None, pos=None):
        """
        This function draws a graph of the communities for a Pyomo model.

        The type_of_graph parameter is used to create either a variable-node graph, constraint-node graph, or
        bipartite graph of the Pyomo model. Then, the nodes are colored based on the communities they are in - which
        is based on the community map (self.community_map). A filename can be provided to save the figure, otherwise
        the figure is illustrated with matplotlib.

        Parameters
        ----------
        type_of_graph: str, optional
            a string that specifies the types of nodes drawn on the model graph, the default is 'constraint'.
            'constraint' draws a graph with constraint nodes,
            'variable' draws a graph with variable nodes,
            'bipartite' draws a bipartite graph (with both constraint and variable nodes)
        filename: str, optional
            a string that specifies a path for the model graph illustration to be saved
        pos: dict, optional
            a dictionary that maps node keys to their positions on the illustration

        Returns
        -------
        fig: matplotlib figure
            the figure for the model graph drawing
        pos: dict
            a dictionary that maps node keys to their positions on the illustration - can be used to create consistent
            layouts for graphs of a given model
        """
        assert type_of_graph in ('bipartite', 'constraint', 'variable'), "Invalid graph type specified: 'type_of_graph=%s' - Valid values: 'bipartite', 'constraint', 'variable'" % type_of_graph
        assert isinstance(filename, (type(None), str)), "Invalid value for filename: 'filename=%s' - filename must be a string" % filename
        if type_of_graph != self.type_of_community_map:
            model_graph, number_component_map, constraint_variable_map = generate_model_graph(self.model, type_of_graph=type_of_graph, with_objective=self.with_objective, weighted_graph=self.weighted_graph, use_only_active_components=self.use_only_active_components)
        else:
            model_graph, number_component_map, constraint_variable_map = (self.graph, self.graph_node_mapping, self.constraint_variable_map)
        component_number_map = ComponentMap(((comp, number) for number, comp in number_component_map.items()))
        numbered_community_map = copy.deepcopy(self.community_map)
        for key in self.community_map:
            numbered_community_map[key] = ([component_number_map[component] for component in self.community_map[key][0]], [component_number_map[component] for component in self.community_map[key][1]])
        if type_of_graph == 'bipartite':
            list_of_node_lists = [list_of_nodes for list_tuple in numbered_community_map.values() for list_of_nodes in list_tuple]
            node_list = [node for sublist in list_of_node_lists for node in sublist]
            color_list = []
            for node in node_list:
                not_found = True
                for community_key in numbered_community_map:
                    if not_found and node in numbered_community_map[community_key][0] + numbered_community_map[community_key][1]:
                        color_list.append(community_key)
                        not_found = False
            if model_graph.number_of_nodes() > 0 and nx.is_connected(model_graph):
                top_nodes = nx.bipartite.sets(model_graph)[1]
            else:
                top_nodes = {node for node in model_graph.nodes() if node in constraint_variable_map}
            if pos is None:
                pos = nx.bipartite_layout(model_graph, top_nodes)
        else:
            position = 0 if type_of_graph == 'constraint' else 1
            list_of_node_lists = list((i[position] for i in numbered_community_map.values()))
            node_list = [node for sublist in list_of_node_lists for node in sublist]
            color_list = []
            for node in node_list:
                not_found = True
                for community_key in numbered_community_map:
                    if not_found and node in numbered_community_map[community_key][position]:
                        color_list.append(community_key)
                        not_found = False
            if pos is None:
                pos = nx.spring_layout(model_graph)
        color_map = plt.cm.get_cmap('viridis', len(numbered_community_map))
        fig = plt.figure()
        nx.draw_networkx_nodes(model_graph, pos, nodelist=node_list, node_size=40, cmap=color_map, node_color=color_list)
        nx.draw_networkx_edges(model_graph, pos, alpha=0.5)
        graph_type = type_of_graph.capitalize()
        community_map_type = self.type_of_community_map.capitalize()
        main_graph_title = '%s graph - colored using %s community map' % (graph_type, community_map_type)
        main_font_size = 14
        plt.suptitle(main_graph_title, fontsize=main_font_size)
        subtitle_naming_dict = {'bipartite': 'Nodes are variables and constraints & Edges are variables in a constraint', 'constraint': 'Nodes are constraints & Edges are common variables', 'variable': 'Nodes are variables & Edges are shared constraints'}
        subtitle_font_size = 11
        plt.title(subtitle_naming_dict[type_of_graph], fontsize=subtitle_font_size)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()
        return (fig, pos)

    def generate_structured_model(self):
        """
        Using the community map and the original model used to create this community map, we will create
        structured_model, which will be based on the original model but will place variables, constraints, and
        objectives into or outside of various blocks (communities) based on the community map.

        Returns
        -------
        structured_model: Block
            a Pyomo model that reflects the nature of the community map
        """
        structured_model = ConcreteModel()
        structured_model.b = Block([0, len(self.community_map) - 1, 1])
        blocked_variable_map = ComponentMap()
        for community_key, community in self.community_map.items():
            _, variables_in_community = community
            for stored_variable in variables_in_community:
                new_variable = Var(domain=stored_variable.domain, bounds=stored_variable.bounds)
                structured_model.b[community_key].add_component(str(stored_variable), new_variable)
                variable_in_new_model = structured_model.find_component(new_variable)
                blocked_variable_map[stored_variable] = blocked_variable_map.get(stored_variable, []) + [variable_in_new_model]
        replace_variables_in_expression_map = dict()
        for community_key, community in self.community_map.items():
            constraints_in_community, _ = community
            for stored_constraint in constraints_in_community:
                for variable_in_stored_constraint in identify_variables(stored_constraint.expr):
                    variable_in_current_block = False
                    for blocked_variable in blocked_variable_map[variable_in_stored_constraint]:
                        if 'b[%d]' % community_key in str(blocked_variable):
                            replace_variables_in_expression_map[id(variable_in_stored_constraint)] = blocked_variable
                            variable_in_current_block = True
                    if not variable_in_current_block:
                        new_variable = Var(domain=variable_in_stored_constraint.domain, bounds=variable_in_stored_constraint.bounds)
                        structured_model.add_component(str(variable_in_stored_constraint), new_variable)
                        variable_in_new_model = structured_model.find_component(new_variable)
                        blocked_variable_map[variable_in_stored_constraint] = blocked_variable_map.get(variable_in_stored_constraint, []) + [variable_in_new_model]
                        replace_variables_in_expression_map[id(variable_in_stored_constraint)] = variable_in_new_model
                if self.with_objective and isinstance(stored_constraint, (_GeneralObjectiveData, Objective)):
                    new_objective = Objective(expr=replace_expressions(stored_constraint.expr, replace_variables_in_expression_map))
                    structured_model.b[community_key].add_component(str(stored_constraint), new_objective)
                else:
                    new_constraint = Constraint(expr=replace_expressions(stored_constraint.expr, replace_variables_in_expression_map))
                    structured_model.b[community_key].add_component(str(stored_constraint), new_constraint)
        if not self.with_objective:
            for objective_function in self.model.component_data_objects(ctype=Objective, active=self.use_only_active_components, descend_into=True):
                for variable_in_objective in identify_variables(objective_function):
                    if structured_model.find_component(str(variable_in_objective)) is None:
                        new_variable = Var(domain=variable_in_objective.domain, bounds=variable_in_objective.bounds)
                        structured_model.add_component(str(variable_in_objective), new_variable)
                        variable_in_new_model = structured_model.find_component(new_variable)
                        blocked_variable_map[variable_in_objective] = blocked_variable_map.get(variable_in_objective, []) + [variable_in_new_model]
                        replace_variables_in_expression_map[id(variable_in_objective)] = variable_in_new_model
                    else:
                        for version_of_variable in blocked_variable_map[variable_in_objective]:
                            if 'b[' not in str(version_of_variable):
                                replace_variables_in_expression_map[id(variable_in_objective)] = version_of_variable
                new_objective = Objective(expr=replace_expressions(objective_function.expr, replace_variables_in_expression_map))
                structured_model.add_component(str(objective_function), new_objective)
        structured_model.equality_constraint_list = ConstraintList(doc='Equality Constraints for the different forms of a given variable')
        for variable, duplicate_variables in blocked_variable_map.items():
            equalities_to_make = combinations(duplicate_variables, 2)
            for variable_1, variable_2 in equalities_to_make:
                structured_model.equality_constraint_list.add(expr=variable_1 == variable_2)
        return structured_model