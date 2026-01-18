import cvxpy.lin_ops.lin_op as lo
class InverseData:
    """Stores data useful for solution retrieval."""

    def __init__(self, problem) -> None:
        varis = problem.variables()
        self.id_map, self.var_offsets, self.x_length, self.var_shapes = InverseData.get_var_offsets(varis)
        self.param_shapes = {}
        self.param_to_size = {lo.CONSTANT_ID: 1}
        self.param_id_map = {}
        offset = 0
        for param in problem.parameters():
            self.param_shapes[param.id] = param.shape
            self.param_to_size[param.id] = param.size
            self.param_id_map[param.id] = offset
            offset += param.size
        self.param_id_map[lo.CONSTANT_ID] = offset
        self.id2var = {var.id: var for var in varis}
        self.id2cons = {cons.id: cons for cons in problem.constraints}
        self.cons_id_map = dict()
        self.constraints = None

    @staticmethod
    def get_var_offsets(variables):
        var_shapes = {}
        var_offsets = {}
        id_map = {}
        vert_offset = 0
        for x in variables:
            var_shapes[x.id] = x.shape
            var_offsets[x.id] = vert_offset
            id_map[x.id] = (vert_offset, x.size)
            vert_offset += x.size
        return (id_map, var_offsets, vert_offset, var_shapes)