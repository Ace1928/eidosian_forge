from Utility import Node
from Algorithm import Algorithm
class DFS(Algorithm):

    def __init__(self, grid):
        super().__init__(grid)

    def recursive_DFS(self, snake, goalstate, currentstate):
        if currentstate.equal(goalstate):
            return self.get_path(currentstate)
        if currentstate in self.explored_set:
            return None
        self.explored_set.append(currentstate)
        neighbors = self.get_neighbors(currentstate)
        for neighbor in neighbors:
            if not self.inside_body(snake, neighbor) and (not self.outside_boundary(neighbor)) and (neighbor not in self.explored_set):
                neighbor.parent = currentstate
                path = self.recursive_DFS(snake, goalstate, neighbor)
                if path != None:
                    return path
        return None

    def run_algorithm(self, snake):
        if len(self.path) != 0:
            path = self.path.pop()
            if self.inside_body(snake, path):
                self.path = []
            else:
                return path
        self.frontier = []
        self.explored_set = []
        self.path = []
        initialstate, goalstate = self.get_initstate_and_goalstate(snake)
        self.frontier.append(initialstate)
        return self.recursive_DFS(snake, goalstate, initialstate)