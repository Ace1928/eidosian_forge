def CollectLabelLevels(tree, levels, level=0, maxDepth=100000000.0):
    if level < maxDepth:
        if not tree.GetTerminal():
            l = tree.GetLabel()
            currLevel = levels.get(l, 100000000.0)
            if level < currLevel:
                levels[l] = level
            for child in tree.GetChildren():
                CollectLabelLevels(child, levels, level + 1, maxDepth)
    return levels