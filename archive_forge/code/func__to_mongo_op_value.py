def _to_mongo_op_value(self, op, value):
    if op == '=':
        return value
    else:
        return {self.INDIVIDUAL_OP_TO_MONGO[op]: value}