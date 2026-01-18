import re
def _operation_tree(self):
    """Build the operation tree from the string representation"""
    i = 0
    level = 0
    stack = []
    current = None

    def _create_operation(args):
        profile_stats = None
        name = args[0].strip()
        args.pop(0)
        if len(args) > 0 and 'Records produced' in args[-1]:
            records_produced = int(re.search('Records produced: (\\d+)', args[-1]).group(1))
            execution_time = float(re.search('Execution time: (\\d+.\\d+) ms', args[-1]).group(1))
            profile_stats = ProfileStats(records_produced, execution_time)
            args.pop(-1)
        return Operation(name, None if len(args) == 0 else args[0].strip(), profile_stats)
    while i < len(self.plan):
        current_op = self.plan[i]
        op_level = current_op.count('    ')
        if op_level == level:
            child = _create_operation(current_op.split('|'))
            if current:
                current = stack.pop()
                current.append_child(child)
            current = child
            i += 1
        elif op_level == level + 1:
            child = _create_operation(current_op.split('|'))
            current.append_child(child)
            stack.append(current)
            current = child
            level += 1
            i += 1
        elif op_level < level:
            levels_back = level - op_level + 1
            for _ in range(levels_back):
                current = stack.pop()
            level -= levels_back
        else:
            raise Exception('corrupted plan')
    return stack[0]