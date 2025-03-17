#!/usr/bin/env python3
# 📊 Eidosian Project Stats Generator
# Analyzes projects and generates stats about the codebase
#
# Author: Lloyd Handyside <ace1928@gmail.com>
# Organization: Neuroforge
# Created: 2025-03-13

import os
import sys
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict

def count_lines_by_extension(file_path):
    """Count lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return len(f.readlines())
    except Exception:
        return 0

def get_file_stats(repo_path):
    """Get statistics about files in the repository."""
    # Escaped literal dictionary inside lambda
    stats = defaultdict(lambda: {"count": 0, "lines": 0})
    total_files = 0
    total_lines = 0
    
    for root, _, files in os.walk(repo_path):
        # Skip hidden directories and node_modules
        if "/." in root or "node_modules" in root or "target" in root or "__pycache__" in root:
            continue
            
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            ext = ext.lstrip('.').lower() or 'no_extension'
            
            lines = count_lines_by_extension(file_path)
            
            stats[ext]["count"] += 1
            stats[ext]["lines"] += lines
            total_files += 1
            total_lines += lines
    
    # Escaped literal dictionary for return
    return {
        "by_extension": dict(stats),
        "total_files": total_files,
        "total_lines": total_lines,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        # Default to the repository root
        script_dir = Path(__file__).parent.absolute()
        repo_path = script_dir.parent.parent
    
    # The following f-strings are meant to be preserved in the generated file,
    # so we escape their curly braces.
    print(f"🔮 Eidosian Project Stats Generator")
    print(f"==================================")
    print(f"Analyzing repository: {repo_path}")
    
    stats = get_file_stats(repo_path)
    
    # Print summary
    print(f"\nRepository Summary:")
    print(f"-------------------")
    print(f"Total files: {stats['total_files']}")
    print(f"Total lines: {stats['total_lines']:,}")
    print("\nFiles by extension:")
    
    # Sort extensions by line count
    sorted_extensions = sorted(
        stats["by_extension"].items(),
        key=lambda x: x[1]["lines"],
        reverse=True
    )
    
    for ext, data in sorted_extensions:
        print(f"  .{ext:<10} {data['count']:5} files, {data['lines']:8,} lines")
    
    # Save to JSON
    output_path = os.path.join(repo_path, "project_stats.json")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✨ Stats saved to {output_path}")
    print(f"Script stats: {stats}")  # Clean, direct logging without complex formatters

if __name__ == "__main__":
    main()
