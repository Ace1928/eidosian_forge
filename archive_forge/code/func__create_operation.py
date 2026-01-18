import re
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