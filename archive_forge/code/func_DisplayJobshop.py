import random
def DisplayJobshop(starts, durations, machines, name):
    """Simple function to display a jobshop solution using plotly."""
    jobs_count = len(starts)
    machines_count = len(starts[0])
    all_machines = range(0, machines_count)
    all_jobs = range(0, jobs_count)
    df = []
    for i in all_jobs:
        for j in all_machines:
            df.append(dict(Task='Resource%i' % machines[i][j], Start=ToDate(starts[i][j]), Finish=ToDate(starts[i][j] + durations[i][j]), Resource='Job%i' % i))
    sorted_df = sorted(df, key=lambda k: k['Task'])
    colors = {}
    cm = ColorManager()
    cm.SeedRandomColor(0)
    for i in all_jobs:
        colors['Job%i' % i] = cm.RandomColor()
    fig = ff.create_gantt(sorted_df, colors=colors, index_col='Resource', title=name, show_colorbar=False, showgrid_x=True, showgrid_y=True, group_tasks=True)
    fig.show()